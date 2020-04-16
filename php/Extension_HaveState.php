<?php


namespace led\Module\Model\Extension;

use App\impl\Log\Log;
use ReflectionClass;

/**
 * Trait HaveState
 * used for add state attributes for model and catch some data
 * private attributes which not will used as auto filled from arrays must start from "__"
 *
 *
 * @package led\Module\Model\Extension
 */
trait HaveState
{
    /**
     * array of attributes which can be settled from array and will automatically returned from array
     * names of private attributes which must be not here must starts from "__"
     * @var array
     */
    protected static $__fillable;

    /**
     * set fillable locked for updates will no dynamic updates
     * if noo automatic feeling or updating __fillable attribute needed set it to true
     * @var bool
     */
    protected $__fillableLocked = false;

    /**
     * starting function on system start
     * @throws \ReflectionException
     */
    protected static function bootHaveState(): void
    {
        static::bootFillable();
    }

    /**
     * set a default properties to fill from arrays at initialisations and at getters and setters
     * @return bool
     * @throws \ReflectionException
     */
    protected static function bootFillable(): bool
    {
        if (count((array)static::$__fillable) > 0) {
            return false;
        }

        $ref = new ReflectionClass(static::class);

        $ownProps = $ref->getProperties();

        $__fillable = array_map(static function ($el) {
            return $el->name;
        }, $ownProps);

        $__fillable = array_filter($__fillable, static function ($el) {
            return $el !== 'table'
                && $el !== 'primary'
                && $el !== 'fillable'
                && strpos($el, '__') !== 0;
        });

        static::$__fillable = array_values($__fillable);

        return true;
    }

    /**
     * get the state of model
     * @return array
     */
    public function getState(): array
    {
        $new = [];

        foreach (static::$__fillable as $key) {
            $new[$key] = $this->get($key);
        }

        return $new;
    }

    /**
     * cast model to array
     * @return array
     */
    public function toArray(): array
    {
        return $this->getState();
    }

    /**
     * get some property from model
     * @param $attribute
     * @param null $default
     * @return static|null
     */
    public function get($attribute, $default = null)
    {
        if (!$this->isAttributeExists($attribute)) {
            return null;
        }

        return $this->{$attribute} ?? $default;
    }

    /**
     * check if attribute existing in model
     * @param $attribute
     * @return bool
     */
    protected function isAttributeExists($attribute): bool
    {
        return property_exists($this, $attribute);
    }

    /**
     * initialize on every model
     * @param $attributes
     * @return HaveState
     */
    protected function initializeHaveState($attributes): self
    {
        return $this->fillFromArray($attributes);
    }

    /**
     * set attributes from value
     * @param $attributes
     * @return $this
     */
    public function fillFromArray($attributes): self
    {
        if (!is_array($attributes)) {
            return $this;
        }

        foreach ($attributes as $key => $attribute) {
            $this->set($key, $attribute);
        }

        return $this;
    }

    /**
     * set needed attribute
     * @param $attribute
     * @param $value
     * @return $this
     */
    public function set($attribute, $value): self
    {
        if (!$this->isAttributeExists($attribute)) {
            return $this;
        }

        $this->{$attribute} = $value;

        return $this;
    }

    /**
     * checks if fillable locked
     * @return bool
     */
    public function isFillableLocked(): bool
    {
        return $this->__fillableLocked;
    }
}